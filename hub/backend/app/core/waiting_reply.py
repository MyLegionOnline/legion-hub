"""Route a user's reply to a task that is waiting for their word.

waiting_reply_safeguard (S1). A direct ``@agent`` build session can leave a
task parked in ``WAITING_REVIEW`` / ``BLOCKED`` with the agent's own question
still hanging ("done, some left to finish — finish?").  By the time the user
answers "finish", the dialogue session has often rotated past its size limit,
so the reply used to be cold-routed to a fresh ChatMaster session that saw a
finished board and just closed itself.

This module gives that reply a home:

* :func:`mark_task_awaiting` / :func:`clear_task_awaiting` — the task-side flag,
  set when a dialogue task enters a waiting state and cleared once the user
  answers or the work is re-dispatched.
* :func:`find_waiting_task` — the most recent still-waiting task in a room,
  within a TTL.
* :func:`classify_reply` — is a short user message an approval, a "carry on",
  or neither.
* :func:`handle_waiting_reply` — the pre-router entry point: approve-and-merge,
  or re-open the task, and tell the room what happened.

Everything is deterministic and best-effort: a phrasing this module does not
recognise falls through to the router, which has its own
``resume_waiting_task`` route (S2).
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Literal
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models.room import Room
from app.models.task import (
    TASK_ORIGIN_DIRECT_SESSION,
    DispatchStatus,
    Task,
    TaskState,
)
from app.models.user import User

logger = logging.getLogger(__name__)

# How long a parked task keeps accepting a bare "finish" as an answer to it.
# After this the reply is treated as a fresh message — the context has moved on.
_AWAITING_TTL = timedelta(hours=float(os.getenv("WAITING_REPLY_TTL_HOURS", "24")))

AwaitingKind = Literal["review", "question", "continuation_offer"]
ReplyIntent = Literal["approve", "continue"]

# Bare confirmations — "yes, ship it". A message counts as an approval only
# when every word in it is an approve-ish token (so "yes, I confirm" matches
# but "yes, but first fix X" does not — that goes to the router).
_APPROVE_WORDS = frozenset(
    """
    finish finalize complete wrap close
    confirm confirmed approve approved accept accepted agree agreed
    yes yep yup yeah sure ok okay go ahead proceed
    right good great perfect fine done excellent awesome suits
    merge ship lgtm
    all everything
    """.split()
)
_APPROVE_STRONG = frozenset(
    """
    finish finalize confirm confirmed approve approved accept accepted
    merge lgtm
    """.split()
)
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# "carry on / finish what you started / do the rest" — an instruction to keep
# working, not a merge approval.
_CONTINUE_RE = re.compile(
    r"(?:"
    r"refine|rework|finish\s+(?:the\s+)?(?:work|task|what\s+(?:you\s+)?started)|"
    r"keep\s+working|push\s+through|bring\s+(?:it\s+)?to\s+(?:the\s+)?end|take\s+it\s+to\s+completion|"
    r"complete\s+the\s+remaining|do\s+the\s+remaining|finish\s+the\s+remaining|"
    r"finish\s+(?:it|the\s+rest|up)|carry\s+on|keep\s+going|continue"
    r")",
    re.IGNORECASE,
)


def waiting_reply_enabled() -> bool:
    """Rollback switch (``WAITING_REPLY_SAFEGUARD=false``)."""
    return str(os.getenv("WAITING_REPLY_SAFEGUARD", "true")).strip().casefold() not in (
        "0", "false", "no", "off", "disabled",
    )


def classify_reply(text: str) -> ReplyIntent | None:
    """Approval, continuation, or neither — for a short mention-free message."""
    body = (text or "").strip()
    if not body or len(body) > 200:
        return None
    if _CONTINUE_RE.search(body):
        return "continue"
    words = [w.lower() for w in _WORD_RE.findall(body)]
    if not words:
        return None
    if all(w in _APPROVE_WORDS for w in words) or (
        len(words) <= 5 and any(w in _APPROVE_STRONG for w in words)
    ):
        return "approve"
    return None


# ── the task-side flag ───────────────────────────────────────────────────────

_WAITING_STATES = frozenset({TaskState.WAITING_REVIEW, TaskState.BLOCKED})


async def mark_task_awaiting(
    task: Task,
    kind: AwaitingKind,
    db: AsyncSession,
    *,
    prompt: str | None = None,
) -> None:
    """Record that *task* is parked waiting for a word from the user.

    Quiet no-op when the task is not in a waiting state — a completed or
    running task is nobody's open question.
    """
    if task.state not in _WAITING_STATES:
        return
    task.awaiting_user_since = datetime.now(timezone.utc)
    task.awaiting_user_kind = kind
    if prompt:
        task.awaiting_user_prompt = prompt.strip()[:2000]
    db.add(task)
    await db.flush()
    logger.info(
        "waiting_reply: task %s parked awaiting user (kind=%s, state=%s)",
        task.id, kind, task.state.value if hasattr(task.state, "value") else task.state,
    )


async def clear_task_awaiting(task: Task, db: AsyncSession) -> None:
    """The user has answered (or the work restarted): drop the flag."""
    if task.awaiting_user_since is None:
        return
    task.awaiting_user_since = None
    task.awaiting_user_kind = None
    task.awaiting_user_prompt = None
    db.add(task)
    await db.flush()


async def find_waiting_task(
    room_id: UUID,
    db: AsyncSession,
    *,
    prefer_session_id: UUID | None = None,
) -> Task | None:
    """The freshest task in *room_id* still waiting for the user, within the TTL.

    When *prefer_session_id* is given (the user's just-closed dialogue), a task
    that came out of that session wins over an older one from elsewhere.
    """
    cutoff = datetime.now(timezone.utc) - _AWAITING_TTL
    rows = (
        await db.execute(
            select(Task)
            .where(
                Task.room_id == room_id,
                Task.awaiting_user_since.isnot(None),
                Task.awaiting_user_since >= cutoff,
                Task.state.in_(tuple(_WAITING_STATES)),
            )
            .order_by(Task.awaiting_user_since.desc())
        )
    ).scalars().all()
    if not rows:
        return None
    if prefer_session_id is not None:
        for task in rows:
            if task.origin_session_id == prefer_session_id:
                return task
    return rows[0]


# ── acting on the reply ──────────────────────────────────────────────────────


async def _post(room_id: UUID, content: str, db: AsyncSession, task_id: UUID | None = None) -> None:
    from app.core.events import emit_event

    await emit_event(
        room_id=room_id,
        event_type="WAITING_REPLY_RESOLVED",
        content=content,
        db=db,
        task_id=task_id,
        metadata={"level": "info"},
    )


async def _approve_task(room: Room, user: User, task: Task, db: AsyncSession) -> None:
    """Fold the reviewed branch into its base — the manual-approve path."""
    from app.core.events import emit_event
    from app.core.goal_progress import maybe_complete_goal
    from app.core.room_git_flow import maybe_open_promotion, merge_task_branch

    if task.state != TaskState.WAITING_REVIEW:
        # A BLOCKED task cannot be "approved" — there is nothing merged to
        # approve. Treat the go-ahead as "carry on" instead.
        await _continue_task(room, user, task, db, "")
        return

    task.state = TaskState.COMPLETED
    db.add(task)
    await db.flush()
    try:
        merge = await merge_task_branch(str(room.id), task, db)
    except Exception as exc:
        logger.warning("waiting_reply: merge failed for task %s: %s", task.id, exc)
        merge = {"ok": False}

    if merge.get("conflict"):
        task.state = TaskState.WAITING_REVIEW
        db.add(task)
        await db.flush()
        await _post(
            room.id,
            f"Could not merge '{task.title}': the branch diverged from the base (manual reconciliation needed). "
            "The task stayed in review.",
            db, task.id,
        )
        return

    if merge.get("ok"):
        task.merge_commit = merge.get("sha")
        db.add(task)
        await db.flush()
    await clear_task_awaiting(task, db)
    await emit_event(
        room_id=room.id,
        event_type="TASK_APPROVED",
        content=f"Task '{task.title}' approved by user.",
        db=db,
        task_id=task.id,
        metadata={"task_id": str(task.id), "via": "waiting_reply"},
    )
    if task.goal_id:
        try:
            await maybe_complete_goal(task.goal_id, room.id, db)
        except Exception as exc:
            logger.warning("waiting_reply: maybe_complete_goal failed: %s", exc)
    try:
        await maybe_open_promotion(str(room.id), db)
    except Exception as exc:
        logger.warning("waiting_reply: maybe_open_promotion failed: %s", exc)
    await _post(
        room.id,
        f"Approved — '{task.title}' merged into the base branch.",
        db, task.id,
    )


async def _continue_task(
    room: Room, user: User, task: Task, db: AsyncSession, instruction: str
) -> None:
    """Re-open the task so the work gets finished.

    A ``direct_session`` task lives in the user ↔ agent dialogue, so continuing
    it means a fresh dialogue turn bound to the task's branch. A planned task
    goes back through the normal rework dispatch.
    """
    from app.core.events import emit_event

    agent_id = task.assigned_agent_id
    if agent_id is None:
        await _post(
            room.id,
            f"No one to continue '{task.title}' — the task has no assignee. "
            "Assign an agent and try again.",
            db, task.id,
        )
        return

    feedback = (instruction or "").strip() or (
        "The user asks to finish what was started. Bring the task to completion "
        "and return it to review."
    )
    prior = (task.awaiting_user_prompt or "").strip()
    await clear_task_awaiting(task, db)

    if task.origin == TASK_ORIGIN_DIRECT_SESSION:
        await _continue_direct_session_task(room, user, task, db, feedback, prior)
        return

    # Planned task: manual rework dispatch.
    from app.core.proxy_session import request_task_dispatch
    from app.core.watchdog import close_open_task_sessions

    await close_open_task_sessions(task.id, "waiting_reply_rework", db)
    task.state = TaskState.ASSIGNED
    task.dispatch_status = DispatchStatus.READY
    task.run_id = None
    task.runtime_state = "idle"
    task.stop_requested_at = None
    task.stop_reason = None
    task.updated_at = datetime.utcnow()
    db.add(task)
    await db.flush()
    try:
        outcome = await request_task_dispatch(
            task, room.id, db,
            resume=True,
            rework_feedback=feedback if not prior else f"{prior}\n\n{feedback}",
            source="waiting_reply",
        )
    except Exception as exc:
        logger.warning("waiting_reply: rework dispatch failed for task %s: %s", task.id, exc)
        await _post(room.id, f"Could not restart '{task.title}': {exc}", db, task.id)
        return
    await emit_event(
        room_id=room.id,
        event_type="TASK_CHANGES_REQUESTED",
        content=f"Task '{task.title}' re-opened by user to finish the remaining work.",
        db=db,
        task_id=task.id,
        metadata={"task_id": str(task.id), "via": "waiting_reply", "dispatch": outcome},
    )
    await _post(room.id, f"Resuming '{task.title}' — finishing what was started.", db, task.id)


async def _continue_direct_session_task(
    room: Room, user: User, task: Task, db: AsyncSession, feedback: str, prior: str
) -> None:
    from app.core.router import RouteDecision
    from app.core.session_work import SNAPSHOT_TASK_ID, SNAPSHOT_WORK_REQUEST
    from app.core.user_sessions import process_user_turn, start_user_session

    # WAITING_REVIEW / BLOCKED → IN_PROGRESS are both allowed transitions.
    if task.state in _WAITING_STATES:
        task.state = TaskState.IN_PROGRESS
        task.updated_at = datetime.utcnow()
        db.add(task)
        await db.flush()

    session = await start_user_session(
        room, user,
        RouteDecision(
            route="single_agent_p2",
            mentioned_agent_ids=[task.assigned_agent_id],
            confidence=1.0,
            reasoning="waiting_reply: resume direct-session task",
        ),
        db,
    )
    # Pin the dialogue to the task's own branch: session_work.ensure_worktree_branch
    # reads SNAPSHOT_TASK_ID via open_dialogue_task and keeps building on task/<id>.
    snap = dict(session.context_snapshot or {})
    snap[SNAPSHOT_TASK_ID] = str(task.id)
    if task.title:
        snap.setdefault(SNAPSHOT_WORK_REQUEST, task.title[:4000])
    session.context_snapshot = snap
    db.add(session)
    await db.flush()

    turn = feedback
    if prior:
        turn = (
            f"You wrote earlier:\n{prior}\n\n"
            f"The user's reply: {feedback}\n\n"
            "Continue on the same task branch and finish the work."
        )
    await _post(
        room.id,
        f"Returning '{task.title}' to @{await _nick(task.assigned_agent_id, db)} — to finish.",
        db, task.id,
    )
    await process_user_turn(session, turn, db)


async def _nick(agent_id: UUID | None, db: AsyncSession) -> str:
    if agent_id is None:
        return "the agent"
    from app.models.agent import Agent

    agent = await db.get(Agent, agent_id)
    return getattr(agent, "nickname", None) or "the agent"


async def handle_waiting_reply(
    room: Room,
    user: User,
    content: str,
    db: AsyncSession,
    *,
    prefer_session_id: UUID | None = None,
    task: Task | None = None,
) -> bool:
    """Pre-router hook: if *content* answers a parked task, act on it.

    Returns True when the reply was consumed (the caller must not route it
    onward), False when there is nothing waiting or the phrasing is not an
    answer.
    """
    if not waiting_reply_enabled():
        return False

    intent = classify_reply(content)
    if intent is None:
        return False

    if task is None:
        task = await find_waiting_task(room.id, db, prefer_session_id=prefer_session_id)
    if task is None:
        return False

    logger.info(
        "waiting_reply: '%s' → intent=%s on task %s (kind=%s)",
        (content or "").strip()[:40], intent, task.id, task.awaiting_user_kind,
    )
    try:
        if intent == "approve":
            await _approve_task(room, user, task, db)
        else:
            await _continue_task(room, user, task, db, content)
    except Exception as exc:
        logger.exception("waiting_reply: handling failed for task %s: %s", task.id, exc)
        return False
    return True
