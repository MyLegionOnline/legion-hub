"""Rename a room's working branch (internal design notes Stage 3).

Several people may bring the same repository into their own rooms; with every
room on ``dev`` they all push to one remote ``dev``. Renaming gives each room
its own integration branch (``core``, ``transport`` …) while ``main`` stays the
shared meeting point behind the promotion gate.

The rename is ``git branch -m`` in workspaced (commits, reflog and a HEAD on
the branch move with it), then ``git_config.working_branch`` in the Hub. Every
other part of the flow already reads the name from the config, so task, goal
and session branches, the promotion, the CI webhook and Pull follow the new
name from the next operation on.

Refused while work is moving through the branch: an open promotion, a goal
gate that is testing or waiting for review, a running task, a branch
divergence under review, an open Pull conflict, or a dialogue turn in flight.
The old branch on the remote is left alone.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import workspaced_client as wsd
from app.core.events import emit_event
from app.core.room_git_flow import PROMOTION_WAITING, get_flow_config
from app.models.goal import RoomGoal
from app.models.promotion import Promotion
from app.models.remote_divergence import OPEN_REMOTE_DIVERGENCE, RemoteDivergence
from app.models.room import Room
from app.models.session.base import Session as SessionRow
from app.models.task import Task, TaskState

logger = logging.getLogger(__name__)

EVENT_TYPE = "WORKING_BRANCH_RENAMED"

_RUNNING_TASK_STATES = (TaskState.IN_PROGRESS, TaskState.PAUSING, TaskState.RESUMING)
_GOAL_GATE_STATES = ("testing", "waiting_review")


class RenameRefused(Exception):
    """The rename was not done. ``status`` is the HTTP status for the caller."""

    def __init__(self, status: int, code: str, message: str, blockers: Optional[list] = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.blockers = blockers or []

    def detail(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "blockers": self.blockers}


@dataclass
class RenameOutcome:
    ok: bool
    changed: bool
    from_branch: str
    to_branch: str
    # The remote already has a branch of the new name; nothing was changed
    # until the caller confirms (``confirm_remote_exists``).
    remote_exists: bool = False
    remote_checked: bool = False
    head: Optional[str] = None
    notes: list[str] = field(default_factory=list)


def _blocker(kind: str, count: int, message: str) -> dict[str, Any]:
    return {"kind": kind, "count": count, "message": message}


async def _count(db: AsyncSession, stmt) -> int:
    return int((await db.execute(stmt)).scalar() or 0)


async def rename_blockers(room_id: UUID, db: AsyncSession) -> list[dict[str, Any]]:
    """What in the room is moving through the working branch right now.

    A promotion in ``conflict`` is not counted: it is final (the next one is
    opened afresh) and already put HEAD back on the working branch.
    """
    from app.core.user_sessions import turn_in_progress

    blockers: list[dict[str, Any]] = []

    n = await _count(db, select(func.count()).select_from(Promotion).where(
        Promotion.room_id == room_id, Promotion.state == PROMOTION_WAITING,
    ))
    if n:
        blockers.append(_blocker(
            "promotion", n, "a promotion to main is waiting for review — approve or reject it first",
        ))

    n = await _count(db, select(func.count()).select_from(RoomGoal).where(
        RoomGoal.room_id == room_id, RoomGoal.review_state.in_(_GOAL_GATE_STATES),
    ))
    if n:
        blockers.append(_blocker(
            "goal_review", n, "a goal is being tested or waits for review — resolve it first",
        ))

    n = await _count(db, select(func.count()).select_from(Task).where(
        Task.room_id == room_id,
        or_(Task.state.in_(_RUNNING_TASK_STATES), Task.divergence_state == "reviewing"),
    ))
    if n:
        blockers.append(_blocker(
            "task", n, "a task is running — wait for it to finish or pause it",
        ))

    n = await _count(db, select(func.count()).select_from(RemoteDivergence).where(
        RemoteDivergence.room_id == room_id, OPEN_REMOTE_DIVERGENCE,
    ))
    if n:
        blockers.append(_blocker(
            "pull_conflict", n, "a Pull conflict is open — resolve it first",
        ))

    live = (await db.execute(select(SessionRow.id).where(
        SessionRow.room_id == room_id, SessionRow.closed_at.is_(None),
    ))).scalars().all()
    n = sum(1 for sid in live if turn_in_progress(sid))
    if n:
        blockers.append(_blocker(
            "dialogue", n, "an agent is answering in a dialogue — wait for the reply",
        ))
    return blockers


def _refusal_from_workspaced(result: dict[str, Any]) -> RenameRefused:
    code = result.get("code")
    message = result.get("error") or result.get("stderr") or "rename failed"
    if result.get("busy"):
        return RenameRefused(409, "busy", "the room's repository is busy — try again in a moment")
    if code == "invalid_name":
        return RenameRefused(422, "invalid_name", message)
    if code in ("branch_exists", "working_branch_mismatch"):
        return RenameRefused(409, code, message)
    return RenameRefused(502, "workspaced_error", message)


async def rename_working_branch(
    room_id: UUID,
    new_name: str,
    db: AsyncSession,
    *,
    access_token: Optional[str] = None,
    confirm_remote_exists: bool = False,
    user_id: Optional[UUID] = None,
) -> RenameOutcome:
    """Rename the room's working branch to ``new_name``.

    Raises :class:`RenameRefused`. Returns ``remote_exists=True`` (and changes
    nothing) when the remote already has ``new_name`` and the caller has not
    confirmed. On a failed config save the git rename is undone.
    """
    from app.core.git_sync import _clone_url

    name = (new_name or "").strip()
    room = await db.get(Room, room_id)
    if room is None:
        raise RenameRefused(404, "not_found", "Room not found")
    old = get_flow_config(room).working_branch
    if not _valid_name(name):
        raise RenameRefused(
            422, "invalid_name",
            f"{name!r} cannot be the working branch: use ASCII letters, digits, '.', '_', "
            "'-' or '/', not 'main' or 'HEAD', and no task/, goal/ or session/ prefix",
        )
    if name == old:
        return RenameOutcome(ok=True, changed=False, from_branch=old, to_branch=name)

    blockers = await rename_blockers(room_id, db)
    if blockers:
        raise RenameRefused(
            409, "work_in_progress",
            "The working branch cannot be renamed while work is moving through it.",
            blockers,
        )

    git_config = dict(room.git_config or {})
    repo_url = _clone_url(str(git_config.get("repo_url") or ""))
    remote_checked = False
    remote_exists = False
    if repo_url:
        try:
            probe = await wsd.remote_branch_exists(str(room_id), repo_url, name, access_token)
        except Exception as exc:  # network / workspaced down: do not guess
            logger.warning("rename_working_branch: remote check failed for room %s: %s", room_id, exc)
            probe = {"ok": False, "error": str(exc)}
        remote_checked = bool(probe.get("ok"))
        remote_exists = remote_checked and bool(probe.get("exists"))
        if remote_exists and not confirm_remote_exists:
            return RenameOutcome(
                ok=False, changed=False, from_branch=old, to_branch=name,
                remote_exists=True, remote_checked=True,
            )

    result = await wsd.rename_working_branch(str(room_id), old, name)
    if not result.get("ok"):
        raise _refusal_from_workspaced(result)

    # Re-read: another save may have landed while git was busy.
    await db.refresh(room)
    # Written here, never from a settings save (SERVER_OWNED_GIT_KEYS). A new
    # mapping, so the JSON column is marked dirty without flag_modified.
    room.git_config = {**(room.git_config or {}), "working_branch": name}
    try:
        await db.commit()
    except Exception:
        await db.rollback()
        logger.exception("rename_working_branch: could not store %s for room %s; undoing", name, room_id)
        try:
            await wsd.rename_working_branch(str(room_id), name, old)
        except Exception:
            logger.exception("rename_working_branch: undo %s → %s failed for room %s", name, old, room_id)
        raise RenameRefused(500, "save_failed", "The new name could not be saved; nothing changed.")

    notes: list[str] = []
    if repo_url:
        notes.append(f"The branch {old} on the remote was left as it is.")
        notes.append(f"CI workflows that run on {old} must be pointed at {name}.")
        if remote_exists:
            notes.append(f"The remote already had a branch {name}; the next push must fit on top of it.")
        if not remote_checked:
            notes.append(f"The remote could not be checked for an existing branch {name}.")
    content = f"Working branch renamed: {old} → {name}."
    if notes:
        content += " " + " ".join(notes)
    try:
        await emit_event(
            room_id=room_id,
            event_type=EVENT_TYPE,
            content=content,
            db=db,
            metadata={
                "from_branch": old,
                "to_branch": name,
                "renamed": bool(result.get("renamed")),
                "created": bool(result.get("created")),
                "remote_branch_exists": remote_exists,
                "by_user_id": str(user_id) if user_id else None,
            },
        )
    except Exception:
        logger.warning("rename_working_branch: event for room %s not emitted", room_id, exc_info=True)

    logger.info("rename_working_branch: room %s %s → %s", room_id, old, name)
    return RenameOutcome(
        ok=True, changed=True, from_branch=old, to_branch=name,
        remote_checked=remote_checked, head=result.get("head"), notes=notes,
    )


def _valid_name(name: str) -> bool:
    """Same rules as workspaced's ``valid_working_branch_name``; checked here
    too so a bad name never reaches the remote probe."""
    import re

    if not name or not re.match(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,99}$", name):
        return False
    if name in ("main", "HEAD") or name.startswith(("task/", "goal/", "session/")):
        return False
    if ".." in name or "//" in name or "@{" in name or name.endswith((".", "/")):
        return False
    return all(part and not part.startswith(".") and not part.endswith(".lock")
               for part in name.split("/"))
