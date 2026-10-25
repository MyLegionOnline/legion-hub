"""
Watchdog for stalled proxy sessions and orphaned tasks (Chat Upgrade stage 4).

Three responsibilities:
  * cleanup_orphaned_sessions — at application startup: proxy sessions left open
    by a previous process (the _run_proxy_loop coroutine no longer exists) are
    closed with reason='orphaned_restart'; their IN_PROGRESS tasks go to BLOCKED
    with a persistent TASK_STALLED event so the user can hit Retry. User-facing
    sessions are closed the same way — nothing resumes them either.
  * run_watchdog_cycle — every WATCHDOG_INTERVAL_SECONDS: open proxy sessions
    with last_activity_at older than SESSION_STALL_MINUTES get an ephemeral
    'session_stalled' status (re-emitted each cycle while stalled); older than
    SESSION_KILL_MINUTES — both sides are closed with reason='timeout' and the
    task goes to BLOCKED + persistent TASK_STALLED. User-facing sessions silent
    for USER_SESSION_IDLE_MINUTES are closed with reason='idle'
    (internal design notes stage 1.4).
  * close_stale_plan_sessions — at startup (orphaned_restart) and every cycle
    (timeout): cm_self_plan decomposition sessions silent for
    PLAN_SESSION_STALE_MINUTES (default 60) — their coroutine died with a
    process kill, so nothing else would ever close them.
  * run_watchdog_loop — infinite loop, started from main.py like log_cleanup.

Every session this module closes goes through flush_session_memory
(internal design notes §4.1) so orphaned_restart/timeout/
idle closes — none of which give the model a chance to call close_session —
still leave a non-empty Agent memory record. block_stalled_task additionally
triggers maybe_refresh_room_memory (§4.2) since a task going BLOCKED is one of
its two terminal-event triggers.

The watchdog reads state from the database (not process memory), so it is
correct with multiple workers.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.events import AgentStatus, emit_event, emit_status
from app.core.session_close import flush_session_memory, maybe_refresh_room_memory
from app.core.user_session_limits import (
    USER_SESSION_KINDS,
    idle_timeout,
    lifecycle_enforced,
    max_age,
    survive_restart,
)
from app.models.message import PROXY_MESSAGE, SESSION_CLOSED, TASK_STALLED, Message
from app.models.session import Session
from app.models.session.cm_agent_proxy import CmAgentProxySession
from app.models.task import Task, TaskState

logger = logging.getLogger(__name__)

_INTERVAL_SECONDS = int(os.getenv("WATCHDOG_INTERVAL_SECONDS", "60"))


def _minutes_env(name: str, default: float) -> timedelta:
    try:
        return timedelta(minutes=float(os.getenv(name, str(default))))
    except ValueError:
        logger.warning("watchdog: invalid %s — using default %s min", name, default)
        return timedelta(minutes=default)


def _stall_threshold() -> timedelta:
    return _minutes_env("SESSION_STALL_MINUTES", 3)


def _kill_threshold() -> timedelta:
    return _minutes_env("SESSION_KILL_MINUTES", 15)


def _plan_stale_threshold() -> timedelta:
    # A decomposition turn is a handful of LLM calls, each stamping
    # last_activity_at (plan_session._touch_activity); an hour of silence
    # means its coroutine is gone, never that it is still thinking.
    return _minutes_env("PLAN_SESSION_STALE_MINUTES", 60)


async def _cm_awaiting_agent_since(
    session: Session, db: AsyncSession
) -> datetime | None:
    """When the agent last handed the chat_master side something still unanswered.

    A cm_agent_proxy pair shares one execution loop (proxy_session._run_proxy_loop):
    it only runs a ChatMaster turn in response to the agent — a mid-task question,
    a stall report, or the completed-work hand-off. So while the agent is working,
    the chat_master side has *nothing to do* and its last_activity_at legitimately
    goes stale. Reporting that as "no activity" (or, past the kill threshold,
    tearing the pair down and taking the agent's live session with it) is a false
    positive.

    Returns the timestamp of the most recent PROXY_MESSAGE for the task when it
    came from the agent (CM owes a reply since then), or None when the CM side is
    deliberately idle — the last message was CM's own, or the pair has not
    exchanged anything yet. The returned datetime is tz-aware (UTC).
    """
    last = (
        await db.execute(
            select(Message)
            .where(
                Message.event_type == PROXY_MESSAGE,
                Message.task_id == session.task_id,
            )
            .order_by(Message.timestamp.desc())
            .limit(1)
        )
    ).scalars().first()
    if last is None or (last.event_metadata or {}).get("sender_type") != "agent":
        return None
    ts = last.timestamp
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


async def _emit_session_closed(session: Session, close_reason: str, db: AsyncSession) -> None:
    side: str = getattr(session, "side", "unknown")
    await emit_event(
        room_id=session.room_id,
        event_type=SESSION_CLOSED,
        content=(
            f"Session {session.id} closed "
            f"(kind={session.kind}, side={side}, reason={close_reason})"
        ),
        db=db,
        metadata={
            "session_id": str(session.id),
            "kind": session.kind,
            "side": side,
            "close_reason": close_reason,
            "summary": None,
        },
    )


async def block_stalled_task(
    task_id, reason: str, content: str, db: AsyncSession, *, extra_meta: dict | None = None
) -> None:
    """Move an IN_PROGRESS task to BLOCKED and emit a persistent TASK_STALLED event."""
    task = (
        await db.execute(select(Task).where(Task.id == task_id))
    ).scalars().first()
    if task is None or task.state != TaskState.IN_PROGRESS:
        return
    # Keep the lifecycle diagnosis and the actual runtime fence in the same
    # transaction.  This avoids a separately fetched ORM row overwriting the
    # task state in a concurrent watchdog cycle.
    task.runtime_state = "stop_requested"
    task.stop_requested_at = datetime.now(timezone.utc)
    task.stop_reason = reason
    task.state = TaskState.BLOCKED
    # A stalled task's dispatch_status is almost always still DISPATCHED (the
    # coroutine that would have flipped it back is exactly what stalled).
    # Leaving it there makes request_task_dispatch's idempotency check treat
    # the task as forever "already active" — reset it so Retry actually works.
    from app.models.task import DispatchStatus

    task.dispatch_status = DispatchStatus.READY
    task.updated_at = datetime.utcnow()
    db.add(task)
    await db.flush()
    await emit_event(
        room_id=task.room_id,
        event_type=TASK_STALLED,
        content=content,
        db=db,
        task_id=task.id,
        metadata={"task_id": str(task.id), "reason": reason, **(extra_meta or {})},
    )
    # §4.2: the task just went BLOCKED — one of the two Room memory triggers.
    await maybe_refresh_room_memory(task.room_id, db, trigger="task_blocked")


async def close_open_task_sessions(
    task_id,
    close_reason: str,
    db: AsyncSession,
    *,
    now: datetime | None = None,
    cancel_local: bool = False,
    include_dialogue: bool = False,
) -> int:
    """Close every still-open proxy session of a task (both sides of the pair).

    Used by the retry endpoint (chat upgrade stage 5) with close_reason='retry'.
    Emits a persistent SESSION_CLOSED event per closed session.
    Returns the number of sessions closed.

    With include_dialogue, open user↔agent sessions bound to the same task are
    closed too. A proxy run is not the only way work happens on a task: a direct
    dialogue with the executor drives the same workspace, and stopping the task
    while that conversation kept running is how a cancelled task went on writing
    files. Only lifecycle stops (cancel/block/forced completion) pass it — a
    retry must not kill the user's conversation.
    """
    now = now or datetime.now(timezone.utc)
    sessions: list[Session] = list(
        (
            await db.execute(
                select(CmAgentProxySession).where(
                    CmAgentProxySession.task_id == task_id,
                    CmAgentProxySession.closed_at.is_(None),
                )
            )
        ).scalars().all()
    )
    if include_dialogue:
        from app.models.session.user_agent import UserAgentSession

        from app.core.session_work import SNAPSHOT_TASK_ID

        # A dialogue that opened its task itself (p2 route, "adhoc" work) keeps
        # the link only in context_snapshot — its task_id column stays NULL, so
        # matching on the column alone left that conversation running after the
        # user cancelled the task.
        sessions.extend(
            (
                await db.execute(
                    select(UserAgentSession).where(
                        or_(
                            UserAgentSession.task_id == task_id,
                            UserAgentSession.context_snapshot[SNAPSHOT_TASK_ID].astext
                            == str(task_id),
                        ),
                        UserAgentSession.closed_at.is_(None),
                    )
                )
            ).scalars().all()
        )
    # Pick up open peers that the task_id filter missed (e.g. a CM side stored
    # without task_id) via the pair link.
    seen = {s.id for s in sessions}
    for session in list(sessions):
        peer_id = session.peer_session_id
        if peer_id and peer_id not in seen:
            peer = (
                await db.execute(
                    select(Session).where(
                        Session.id == peer_id, Session.closed_at.is_(None)
                    )
                )
            ).scalars().first()
            if peer is not None:
                sessions.append(peer)
                seen.add(peer.id)

    if not sessions:
        return 0

    for session in sessions:
        session.closed_at = now
        session.close_reason = close_reason
        db.add(session)
    await db.flush()

    for session in sessions:
        await flush_session_memory(session, db, reason=close_reason)
        await _emit_session_closed(session, close_reason, db)

    if cancel_local or close_reason in {"retry", "cancelled", "timeout"}:
        from app.core.task_runtime import cancel_local_proxy_task

        cancel_local_proxy_task(task_id)

    return len(sessions)


# ---------------------------------------------------------------------------
# Startup: orphaned sessions (server restarted while proxy loops were running)
# ---------------------------------------------------------------------------


async def _finalize_dialogue_work(session: Session, db: AsyncSession, *, reason: str) -> None:
    """Best-effort Gate 1 for a dialogue session the watchdog is closing.

    internal design notes stage 4.2: neither of the paths that call
    this (idle, orphaned_restart) gives the model a chance to call
    close_session, so nothing else would ever run Gate 1 on what the dialogue's
    last turn wrote. The room's container may not even be reachable right now
    (orphaned_restart in particular runs right after a server restart) — a
    failure here just leaves the branch where it is; sweep_orphan_session_work
    picks it up when the room's next direct session opens (stage 4.3). No-op
    for non-dialogue session kinds (session_work.finalize_session_work checks).
    """
    try:
        from app.core.session_work import finalize_session_work

        await finalize_session_work(session, db, reason=reason)
    except Exception as exc:
        logger.warning(
            "watchdog: could not finalize dialogue work for session %s (%s): %s",
            session.id, reason, exc,
        )


async def _close_user_sessions(
    sessions: list[Session], close_reason: str, now: datetime, db: AsyncSession
) -> int:
    """Close user-facing sessions and announce each one in its room.

    Neither of these paths (orphaned_restart, idle) gives the model a chance to
    call close_session, so flush_session_memory (§4.1) writes Agent memory
    itself — reusing session.summary when set, else a cheap/deterministic note.
    """
    if not sessions:
        return 0
    for session in sessions:
        session.closed_at = now
        session.close_reason = close_reason
        db.add(session)
    await db.flush()
    for session in sessions:
        await flush_session_memory(session, db, reason=close_reason)
        await _finalize_dialogue_work(session, db, reason=close_reason)
        await _save_unfinished_dialogue_draft(session, close_reason, db)
        await _emit_session_closed(session, close_reason, db)
    return len(sessions)


async def _save_unfinished_dialogue_draft(
    session: Session, close_reason: str, db: AsyncSession
) -> None:
    """L1 draft for a dialogue the watchdog cut off before it produced work.

    internal design notes Stage 5 saved drafts only on the paths where
    the model itself closed the session. idle / orphaned_restart are exactly
    the "the user walked away mid-gathering" cases — without a draft the next
    ChatMaster session has no candidate to ``merge_draft`` and restarts the
    requirement gathering from zero. A dialogue that already formed a goal or
    created tasks has its own archive; it is not "unfinished".
    """
    try:
        from app.models.goal import RoomGoal

        produced_task = (
            await db.execute(
                select(Task.id).where(Task.origin_session_id == session.id).limit(1)
            )
        ).first()
        produced_goal = (
            await db.execute(
                select(RoomGoal.id).where(RoomGoal.origin_session_id == session.id).limit(1)
            )
        ).first()
        if produced_task is not None or produced_goal is not None:
            return

        from app.core.user_sessions import _maybe_save_dialogue_draft

        await _maybe_save_dialogue_draft(session, db, close_reason=close_reason)
    except Exception as exc:
        logger.warning(
            "watchdog: could not save dialogue draft for session %s (%s): %s",
            session.id, close_reason, exc,
        )


async def cleanup_orphaned_sessions(db: AsyncSession, *, now: datetime | None = None) -> int:
    """Close every open proxy session and block its IN_PROGRESS task.

    Called once at startup, before the watchdog loop: any proxy session that is
    still open in the database at this point has no living _run_proxy_loop
    coroutine — the server was restarted mid-conversation.  User-facing
    sessions are different: they have no loop, so a restart orphans nothing.
    Only the stale ones (past the idle timeout or max age) are closed here;
    ``USER_SESSION_SURVIVE_RESTART=false`` closes them all, as stage 1.4 did.
    Returns the number of sessions closed.
    """
    now = now or datetime.now(timezone.utc)
    sessions = (
        await db.execute(
            select(CmAgentProxySession).where(CmAgentProxySession.closed_at.is_(None))
        )
    ).scalars().all()

    if sessions:
        for session in sessions:
            session.closed_at = now
            session.close_reason = "orphaned_restart"
            db.add(session)
        await db.flush()

        for session in sessions:
            await flush_session_memory(session, db, reason="orphaned_restart")
            await _emit_session_closed(session, "orphaned_restart", db)

        for task_id in {s.task_id for s in sessions if s.task_id}:
            await block_stalled_task(
                task_id,
                "orphaned_restart",
                "The server was restarted — the task was orphaned. Press Retry to restart it.",
                db,
            )
        logger.info("watchdog: closed %d orphaned proxy sessions at startup", len(sessions))

    closed_user = 0
    if lifecycle_enforced():
        user_query = select(Session).where(
            Session.kind.in_(USER_SESSION_KINDS),
            Session.closed_at.is_(None),
        )
        if survive_restart():
            # A user session has no loop a restart could kill — only the stale
            # ones (the same idle / age limits the running server applies) are
            # closed here. A young one stays open, so the user's next message
            # is offered to the router as continue_session instead of starting
            # a cold ChatMaster that has lost the conversation.
            activity = func.coalesce(Session.last_activity_at, Session.started_at)
            user_query = user_query.where(
                (activity < now - idle_timeout()) | (Session.started_at < now - max_age())
            )
        user_sessions: list[Session] = list(
            (await db.execute(user_query)).scalars().all()
        )
        closed_user = await _close_user_sessions(
            user_sessions, "orphaned_restart", now, db
        )
        if closed_user:
            logger.info(
                "watchdog: closed %d orphaned user sessions at startup", closed_user
            )

    closed_plan = await close_stale_plan_sessions(db, reason="orphaned_restart", now=now)

    return len(sessions) + closed_user + closed_plan


async def close_stale_plan_sessions(
    db: AsyncSession, *, reason: str, now: datetime | None = None
) -> int:
    """Close cm_self_plan sessions silent for PLAN_SESSION_STALE_MINUTES.

    plan_session closes every run it can observe ending (including errors and
    cancellation), but a process kill mid-turn leaves the row open with
    nothing to resume it. The threshold — not "everything open" — keeps this
    safe with several workers: a live turn stamps last_activity_at on every
    LLM step, so only a dead one ever crosses it.
    """
    now = now or datetime.now(timezone.utc)
    activity = func.coalesce(Session.last_activity_at, Session.started_at)
    sessions: list[Session] = list(
        (
            await db.execute(
                select(Session).where(
                    Session.kind == "cm_self_plan",
                    Session.closed_at.is_(None),
                    activity < now - _plan_stale_threshold(),
                )
            )
        ).scalars().all()
    )
    if not sessions:
        return 0
    for session in sessions:
        session.closed_at = now
        session.close_reason = reason
        db.add(session)
    await db.flush()
    for session in sessions:
        await _emit_session_closed(session, reason, db)
    logger.info(
        "watchdog: closed %d stale cm_self_plan session(s) (reason=%s)",
        len(sessions), reason,
    )
    return len(sessions)


async def close_idle_user_sessions(db: AsyncSession, *, now: datetime | None = None) -> int:
    """Close user-facing sessions silent for USER_SESSION_IDLE_MINUTES (1.4).

    A user session has no loop of its own: it lives between HTTP/WS turns, so
    nothing but this cycle can ever end an abandoned conversation.  The next
    message from the user then starts a fresh session instead of appending to a
    transcript that is days old.
    """
    if not lifecycle_enforced():
        return 0
    now = now or datetime.now(timezone.utc)
    cutoff = now - idle_timeout()
    # last_activity_at is written on every user turn; started_at covers a session
    # that was created but never used.
    activity = func.coalesce(Session.last_activity_at, Session.started_at)
    sessions: list[Session] = list(
        (
            await db.execute(
                select(Session).where(
                    Session.kind.in_(USER_SESSION_KINDS),
                    Session.closed_at.is_(None),
                    activity < cutoff,
                )
            )
        ).scalars().all()
    )
    closed = await _close_user_sessions(sessions, "idle", now, db)
    if closed:
        logger.info(
            "watchdog: closed %d idle user session(s) (idle threshold %s)",
            closed, idle_timeout(),
        )
    return closed


async def startup_orphan_cleanup(engine) -> None:
    """Run cleanup_orphaned_sessions in its own DB session; never break startup."""
    from sqlalchemy.orm import sessionmaker

    try:
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with async_session() as db:
            await cleanup_orphaned_sessions(db)
            await db.commit()
    except Exception as exc:
        logger.warning("watchdog: orphan cleanup at startup failed: %s", exc)


# ---------------------------------------------------------------------------
# Periodic cycle: stalled and dead sessions
# ---------------------------------------------------------------------------


async def _kill_stalled_pair(
    session: CmAgentProxySession,
    now: datetime,
    handled: set,
    db: AsyncSession,
) -> None:
    """Close both sides of a proxy pair with reason='timeout' and block the task."""
    to_close: list[Session] = [session]
    if session.peer_session_id:
        peer = (
            await db.execute(
                select(Session).where(
                    Session.id == session.peer_session_id,
                    Session.closed_at.is_(None),
                )
            )
        ).scalars().first()
        if peer is not None:
            to_close.append(peer)
            handled.add(peer.id)

    for s in to_close:
        s.closed_at = now
        s.close_reason = "timeout"
        db.add(s)
    await db.flush()

    for s in to_close:
        await flush_session_memory(s, db, reason="timeout")
        await _emit_session_closed(s, "timeout", db)

    last_seen = session.last_activity_at or session.started_at
    stalled_for_s = int((now - last_seen).total_seconds())
    stalled_min = stalled_for_s // 60
    await block_stalled_task(
        session.task_id,
        "timeout",
        f"No activity for {stalled_min} min — the session was closed by timeout. "
        "Press Retry to restart the task.",
        db,
        extra_meta={"stalled_for_s": stalled_for_s},
    )
    # Closing the rows is the durable fence; cancel the in-process worker as
    # well so a provider request cannot continue producing late tool calls.
    from app.core.task_runtime import cancel_local_proxy_task

    cancel_local_proxy_task(session.task_id)
    logger.warning(
        "watchdog: killed stalled proxy pair (session=%s, task=%s, stalled_for=%ss)",
        session.id, session.task_id, stalled_for_s,
    )


async def block_stalled_legacy_tasks(db: AsyncSession, *, now: datetime | None = None) -> int:
    """Recover tasks stuck IN_PROGRESS by the legacy `approve_and_start` pipeline.

    `approve_and_start` (chat_master.py) runs its agent loop synchronously
    inside the HTTP request that confirmed the plan — it never creates a
    CmAgentProxySession and never mints a run_id, so it is entirely invisible
    to the session-based checks in run_watchdog_cycle/cleanup_orphaned_sessions
    above. If the worker dies mid-loop, the task is left IN_PROGRESS forever:
    no session to expire, no run_id to fence, no way for the user to Retry it
    (internal design notes, part 3, stage 3 — approve_and_start migration
    strategy: keep the live legacy pipeline as-is rather than risk rewriting
    a core, synchronous UX path, but stop leaving its failures invisible).

    Its loop is bounded (_MAX_AGENT_TURNS=8, one LLM call per turn, no tool
    execution), so it normally finishes for one task well inside the kill
    threshold — the same threshold used to kill genuinely stalled proxy
    sessions below is a safe staleness signal here too. `run_id IS NULL` is
    what distinguishes an untracked legacy execution from a modern claim that
    merely hasn't flushed a session row yet.
    """
    now = now or datetime.now(timezone.utc)
    kill_cutoff = (now - _kill_threshold()).replace(tzinfo=None)

    stale_ids = (
        await db.execute(
            select(Task.id).where(
                Task.state == TaskState.IN_PROGRESS,
                Task.run_id.is_(None),
                Task.updated_at < kill_cutoff,
            )
        )
    ).scalars().all()
    for task_id in stale_ids:
        await block_stalled_task(
            task_id,
            "legacy_pipeline_stalled",
            "No updates on the task for longer than usual — execution may have "
            "been interrupted. Press Retry to restart it.",
            db,
        )
    if stale_ids:
        logger.warning(
            "watchdog: blocked %d stalled legacy-pipeline task(s): %s",
            len(stale_ids), stale_ids,
        )
    return len(stale_ids)


async def run_watchdog_cycle(db: AsyncSession, *, now: datetime | None = None) -> None:
    """One watchdog pass: emit session_stalled statuses, kill dead sessions."""
    now = now or datetime.now(timezone.utc)
    stall_cutoff = now - _stall_threshold()
    kill_cutoff = now - _kill_threshold()

    await block_stalled_legacy_tasks(db, now=now)
    await close_idle_user_sessions(db, now=now)
    await close_stale_plan_sessions(db, reason="timeout", now=now)

    # Fall back to started_at: a session whose loop crashed before the first
    # commit has last_activity_at = NULL and must not be invisible forever.
    activity = func.coalesce(
        CmAgentProxySession.last_activity_at, CmAgentProxySession.started_at
    )
    stalled = (
        await db.execute(
            select(CmAgentProxySession).where(
                CmAgentProxySession.closed_at.is_(None),
                # A connector-executed session has
                # its own offline timeout (core/connector_session.py,
                # CONNECTOR_DISPATCH_TIMEOUT_MINUTES, default 15 min) anchored
                # on the dispatch frame, not on last_activity_at — there is no
                # backchannel yet (Stage 5) to keep that column fresh while
                # the remote agent genuinely works, so this generic 3-min
                # stall / 15-min kill check would misfire on every one of
                # them.
                CmAgentProxySession.executor != "connector",
                activity < stall_cutoff,
            )
        )
    ).scalars().all()
    if not stalled:
        return

    from app.core.ask_user_registry import is_session_waiting

    handled: set = set()
    for session in stalled:
        if session.id in handled:
            continue
        handled.add(session.id)

        # Session is deliberately idle — it is waiting for a user answer via ask_user.
        # Refreshing last_activity_at here is not necessary; the session will update
        # it itself once the user answers and the tool loop resumes.
        if is_session_waiting(session.id):
            logger.debug(
                "watchdog: session %s is waiting for user answer — skipping stale check",
                session.id,
            )
            continue

        last_seen = session.last_activity_at or session.started_at

        # The chat_master side of a proxy pair is idle *by design* while the
        # agent works — the shared loop only schedules a CM turn on an agent
        # question / stall / hand-off. Start its stall clock from the moment the
        # agent actually handed it something unanswered; until then it is
        # waiting, not hung, and must not be flagged or killed.
        if getattr(session, "side", None) == "chat_master":
            owed_since = await _cm_awaiting_agent_since(session, db)
            if owed_since is None or owed_since >= stall_cutoff:
                # Nothing pending, or the agent handed it over only moments ago
                # (the CM turn's own llm_request_sent status covers that case) —
                # surface a calm "waiting for the agent", never a stall.
                if owed_since is None:
                    await emit_status(
                        session.room_id,
                        AgentStatus.WAITING_PEER,
                        agent_id=session.agent_id,
                        session_id=session.id,
                        task_id=session.task_id,
                        extra={"side": "chat_master", "waiting_for": "agent"},
                    )
                continue
            last_seen = owed_since

        if last_seen < kill_cutoff:
            await _kill_stalled_pair(session, now, handled, db)
        else:
            await emit_status(
                session.room_id,
                AgentStatus.SESSION_STALLED,
                agent_id=session.agent_id,
                session_id=session.id,
                task_id=session.task_id,
                extra={
                    "stalled_for_s": int((now - last_seen).total_seconds()),
                    "side": session.side,
                },
            )


async def run_watchdog_loop(engine) -> None:
    """Infinite loop; intended to run as a background asyncio task."""
    from sqlalchemy.orm import sessionmaker

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    while True:
        try:
            await asyncio.sleep(_INTERVAL_SECONDS)
            async with async_session() as db:
                await run_watchdog_cycle(db)
                await db.commit()
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning("watchdog loop error: %s", exc)
