"""Resume a direct chat when the background process its agent waits for exits.

A long job (a build, a render, a big download) does not fit the agent's turn:
polling burns tool-call steps and a turn cannot outlive its limits. Instead the
agent starts the process with ``bash(background=true, wake_on_exit=true)``, tells
the user what it is waiting for and ends the turn. This module — called from the
workspace reconciler's loop — notices the exit and runs ONE new turn in the
owning session (or in a fresh session when the old one was closed meanwhile),
fed a language-neutral system message with the exit code and the log tail.

Only direct (``user_agent_*``) sessions are woken: a dispatched task's turn
ending is a dialogue with the ChatMaster, so tasks use ``bash_wait`` instead.

Multi-worker safe: a row is claimed with a conditional UPDATE of ``woken_at``,
so exactly one worker resumes it.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectin_polymorphic

logger = logging.getLogger(__name__)

# A process still running after this long stops being waited for.
MAX_WAIT_HOURS = float(os.getenv("BG_WAKE_MAX_HOURS", "12"))
_LOG_TAIL_LINES = 40
_USER_AGENT_KINDS = ("user_agent_p1", "user_agent_p2")

# Strong references to in-flight resume turns (a bare create_task can be GC'd).
_inflight: set[asyncio.Task] = set()


def render_wake_message(
    *, bg_id: str, cmd: str, exit_code: int | None, reason: str | None,
    note: str | None, log_tail: str,
) -> str:
    """The system message the resumed agent sees. English on purpose: it is a
    prompt, not UI text, and the agent answers in the user's own language."""
    if exit_code is not None:
        outcome = f"exit_code: {exit_code}"
    else:
        outcome = f"outcome: {reason or 'unknown'}"
    return "\n".join(filter(None, [
        "[system] The background process you were waiting for has finished. "
        "This is not a message from the user.",
        f"bg_id: {bg_id}",
        f"command: {cmd[:500]}",
        outcome,
        f"your note: {note}" if note else "",
        f"\nlog (last {_LOG_TAIL_LINES} lines):\n{log_tail}" if log_tail else "",
        "\nContinue the work you were doing for the user. If the process failed, "
        "diagnose and fix it. Reply in the language the user writes in.",
    ]))


async def _read_exit_code(room_id: str, exit_path: str) -> int | None:
    from app.core import workspaced_client as wsd

    try:
        raw = await wsd.read_workspace_file(room_id, exit_path)
    except Exception:
        return None
    try:
        return int(raw.strip()) if raw is not None else None
    except ValueError:
        return None


async def _log_tail(room_id: str, log_path: str, bg_id: str) -> str:
    from app.core import workspaced_client as wsd
    from app.core.secrets.room_masking import bg_secrets, mask_room_secrets

    try:
        content = await wsd.read_workspace_file(room_id, log_path)
    except Exception:
        return ""
    tail = "\n".join((content or "").splitlines()[-_LOG_TAIL_LINES:])
    return mask_room_secrets(tail, bg_secrets(bg_id))


def _turn_in_progress(session_id: str) -> bool:
    from app.core.user_sessions import _ACTIVE_TURNS

    try:
        return UUID(session_id) in _ACTIVE_TURNS
    except ValueError:
        return False


async def wake_finished_processes(session_factory: Any) -> int:
    """One sweep: resume the sessions whose awaited process has exited.

    ``session_factory`` is an ``AsyncSession`` maker — each resumed turn runs
    in its own session, detached, so a long turn never blocks the sweep.
    Returns how many were claimed.
    """
    from app.models.workspace import Workspace, WorkspaceBgProc

    now = datetime.now(timezone.utc)
    claimed = 0
    async with session_factory() as db:
        rows = (
            await db.execute(
                select(WorkspaceBgProc, Workspace)
                .join(Workspace, Workspace.id == WorkspaceBgProc.workspace_id)
                .where(
                    WorkspaceBgProc.wake_on_exit.is_(True),
                    WorkspaceBgProc.woken_at.is_(None),
                )
            )
        ).all()
        # Plain values: the ORM rows must not be touched after the commits below.
        pending = [
            {
                "id": p.id, "bg_id": p.bg_id, "session_id": p.session_id,
                "cmd": p.cmd, "log_path": p.log_path, "note": p.wake_note,
                "status": p.status, "exit_code": p.exit_code,
                "created_at": p.created_at,
                "room_id": ws.room_id, "ws_status": ws.status,
            }
            for p, ws in rows
        ]
        await db.commit()

        for item in pending:
            if _turn_in_progress(item["session_id"]):
                continue  # the agent has not ended its turn yet
            room_id = str(item["room_id"])
            exit_code = item["exit_code"]
            if exit_code is None:
                exit_code = await _read_exit_code(room_id, f"{item['log_path']}.exit")
            reason: str | None = None
            if exit_code is None:
                created = item["created_at"]
                if created is not None and created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                if item["status"] in ("failed", "killed"):
                    reason = f"process {item['status']} without an exit code"
                elif item["ws_status"] != "running":
                    reason = "the workspace stopped before the process finished"
                elif created is not None and now - created > timedelta(hours=MAX_WAIT_HOURS):
                    reason = f"gave up waiting after {MAX_WAIT_HOURS:g} h; process still running"
                else:
                    continue  # still running

            new_status = (
                "done" if exit_code == 0 else "failed"
            ) if exit_code is not None else item["status"]
            claim = await db.execute(
                update(WorkspaceBgProc)
                .where(WorkspaceBgProc.id == item["id"], WorkspaceBgProc.woken_at.is_(None))
                .values(
                    woken_at=now,
                    exit_code=exit_code,
                    status=new_status if item["status"] == "running" else item["status"],
                )
            )
            await db.commit()
            if claim.rowcount != 1:
                continue  # another worker took it
            claimed += 1
            message = render_wake_message(
                bg_id=item["bg_id"], cmd=item["cmd"], exit_code=exit_code, reason=reason,
                note=item["note"],
                log_tail=await _log_tail(room_id, item["log_path"], item["bg_id"]),
            )
            task = asyncio.create_task(
                _resume(session_factory, item["session_id"], item["room_id"],
                        item["bg_id"], exit_code, message)
            )
            _inflight.add(task)
            task.add_done_callback(_inflight.discard)
    return claimed


async def _resume(
    session_factory: Any, session_id: str, room_id: UUID, bg_id: str,
    exit_code: int | None, message: str,
) -> None:
    """Run the wake turn in the owning session (or a fresh one)."""
    try:
        async with session_factory() as db:
            await _resume_turn(db, session_id, room_id, bg_id, exit_code, message)
            await db.commit()
    except Exception:
        logger.exception("bg_wake: resuming session %s for %s failed", session_id, bg_id)


async def _resume_turn(
    db: AsyncSession, session_id: str, room_id: UUID, bg_id: str,
    exit_code: int | None, message: str,
) -> None:
    from app.core.events import emit_event
    from app.core.work_gates import refuse_room_work
    from app.core.router import RouteDecision
    from app.core.user_sessions import process_user_turn, start_user_session
    from app.models.room import Room
    from app.models.session import SESSION_SUBTYPES, Session
    from app.models.user import User

    try:
        sid = UUID(session_id)
    except ValueError:
        return
    session = (
        await db.execute(
            select(Session)
            .where(Session.id == sid)
            .options(selectin_polymorphic(Session, SESSION_SUBTYPES))
        )
    ).scalars().first()
    if session is None or session.kind not in _USER_AGENT_KINDS:
        return
    room = await db.get(Room, room_id)
    user = await db.get(User, session.user_id)
    if room is None or user is None:
        return
    if await refuse_room_work(room.id, db):
        return  # out of hours / frozen room: the notice is already posted

    target = session
    if not session.is_open:
        target = await start_user_session(
            room, user,
            RouteDecision(
                route="single_agent_p2",
                mentioned_agent_ids=[session.agent_id],
                confidence=1.0,
                reasoning="bg_wake: awaited background process finished",
            ),
            db,
        )
    await emit_event(
        room_id=room.id,
        event_type="BG_PROCESS_FINISHED",
        content=(
            f"Background process finished (exit code {exit_code}); the agent is resuming."
            if exit_code is not None
            else "Background process ended without an exit code; the agent is resuming."
        ),
        db=db,
        sender_agent_id=session.agent_id,
        metadata={
            "code": "bg_process_finished", "bg_id": bg_id,
            "exit_code": exit_code, "session_id": str(target.id),
        },
    )
    await process_user_turn(target, message, db)
