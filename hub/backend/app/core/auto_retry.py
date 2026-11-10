"""Policy for retrying interrupted task sessions."""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Literal
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.events import emit_event
from app.core.loop_guard import STALL_RESTART_THRESHOLD
from app.models.message import (
    TASK_AUTO_RETRY_EXHAUSTED,
    TASK_AUTO_RETRY_SCHEDULED,
    TASK_AUTO_RETRY_SKIPPED,
)
from app.models.room import Room
from app.models.task import Task, TaskState
from app.models.workspace import Workspace, WorkspaceUsage

logger = logging.getLogger(__name__)

RestartMode = Literal["retry", "resume"]

_AUTO_RETRY_ACTIONS: dict[str, RestartMode] = {
    "provider_error": "retry",
    "timeout": "retry",
    "orphaned_restart": "retry",
    "tool_calls": "resume",
    "context_window": "resume",
}
_BACKOFF_SECONDS = (30, 120, 480)


def auto_retry_action(reason: str, http_status: int | None = None) -> RestartMode | None:
    """Return the automatic restart action for an eligible interruption."""
    if reason == "provider_error" and http_status is not None:
        if http_status != 429 and not 500 <= http_status < 600:
            return None
    return _AUTO_RETRY_ACTIONS.get(reason)


def backoff_seconds(attempt: int, retry_after: float | None = None) -> float:
    """Delay a one-based attempt, honoring Retry-After in seconds when supplied."""
    if attempt < 1:
        raise ValueError("Attempt must be at least 1")
    base_delay = _BACKOFF_SECONDS[min(attempt, len(_BACKOFF_SECONDS)) - 1]
    delay = base_delay * random.uniform(0.8, 1.2)
    return max(delay, retry_after or 0)


def retry_after_seconds(value: str | None) -> float | None:
    """Read either form of the HTTP Retry-After header."""
    if not value:
        return None
    value = value.strip()
    try:
        if value.isascii() and value.isdecimal():
            delay = int(value)
            # An unrepresentable date is an invalid header.
            if delay < (datetime.max.replace(tzinfo=timezone.utc) - datetime.now(timezone.utc)).total_seconds():
                return float(delay)
            return None
        retry_at = parsedate_to_datetime(value)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
    except (ValueError, OverflowError):
        return None


def _active_room_policy(room_id: UUID):
    billing_active = (
        select(WorkspaceUsage.id)
        .join(Workspace, Workspace.id == WorkspaceUsage.workspace_id)
        .where(
            Workspace.room_id == Room.id,
            WorkspaceUsage.stopped_at.is_(None),
            Workspace.disk_stopped_at.is_(None),
            Workspace.archived_at.is_(None),
            Workspace.status.notin_(("draining", "stopping")),
        )
        .exists()
    )
    return select(Room.auto_retry_max).where(
        Room.id == room_id,
        Room.auto_retry_enabled.is_(True),
        Room.frozen_at.is_(None),
        Room.deletion_started_at.is_(None),
        billing_active,
    )


async def auto_retry_policy(room_id: UUID, db: AsyncSession) -> tuple[int | None, bool]:
    """Return ``(limit, contended)`` for a room that is open and billed.

    ``limit`` is None when the room does not allow an automatic attempt now.
    Callers already hold the task, so a concurrent room update is skipped
    instead of waited for: a settings update may itself be clearing task rows.
    ``contended`` is True when the room row was locked and an unlocked read
    says the room would allow the attempt; ``limit`` then carries that read.
    A due attempt re-checks the policy before it runs.
    """
    limit = (
        await db.execute(
            _active_room_policy(room_id).with_for_update(of=Room, read=True, skip_locked=True)
        )
    ).scalar_one_or_none()
    if limit is not None:
        return limit, False
    unlocked = (await db.execute(_active_room_policy(room_id))).scalar_one_or_none()
    return unlocked, unlocked is not None


async def clear_room_auto_retries(room_id: UUID, db: AsyncSession) -> None:
    """Remove pending attempts as part of the room's settings or stop transaction."""
    await db.execute(
        update(Task)
        .where(Task.room_id == room_id, Task.auto_retry_at.is_not(None))
        .values(auto_retry_at=None, auto_retry_reason=None)
    )


@dataclass
class AutoRetryNotice:
    """A room event about the automatic retry policy.

    Scheduling happens in the transaction that blocks the task, before the
    interruption is announced. Callers emit the notice after that
    announcement, so the room history reads in order: what stopped, then what
    the Hub does about it.
    """

    event_type: str
    content: str
    metadata: dict = field(default_factory=dict)

    async def emit(self, task: Task, db: AsyncSession) -> None:
        await emit_event(
            room_id=task.room_id,
            event_type=self.event_type,
            content=self.content,
            db=db,
            task_id=task.id,
            metadata=self.metadata,
        )


async def emit_notice(notice: AutoRetryNotice | None, task: Task, db: AsyncSession) -> None:
    if notice is not None:
        await notice.emit(task, db)


def exhausted_notice(task: Task, reason: str) -> AutoRetryNotice:
    return AutoRetryNotice(
        TASK_AUTO_RETRY_EXHAUSTED,
        f"Auto-retry gave up after {task.auto_retry_count} attempts: {reason}. "
        "Press Retry or Resume.",
        {"reason": reason, "attempt": task.auto_retry_count},
    )


SKIP_NO_PROGRESS = "no_progress"
SKIP_ROOM_INACTIVE = "room_inactive"
SKIP_DISPATCH_REFUSED = "dispatch_refused"

_SKIP_TEXT = {
    SKIP_NO_PROGRESS: "the last sessions ended without progress",
    SKIP_ROOM_INACTIVE: "the room is stopped or frozen, or auto-retry is off",
    SKIP_DISPATCH_REFUSED: "the task cannot start",
}


def skipped_notice(
    task: Task, reason: str, skip_reason: str, detail: str | None = None,
) -> AutoRetryNotice:
    """Explain why no attempt was scheduled, or why a due attempt did not run."""
    text = _SKIP_TEXT[skip_reason]
    if detail:
        text = f"{text}: {detail}"
    metadata = {"reason": reason, "skip_reason": skip_reason}
    if detail:
        metadata["detail"] = detail
    return AutoRetryNotice(
        TASK_AUTO_RETRY_SKIPPED,
        f'Auto-retry skipped for task "{task.title}": {text}. Press Retry or Resume.',
        metadata,
    )


async def schedule_auto_retry(
    task: Task,
    reason: str,
    http_status: int | None,
    db: AsyncSession,
    *,
    retry_after: float | None = None,
) -> AutoRetryNotice | None:
    """Schedule recovery in the same transaction that blocks the locked task.

    Returns the room event to emit once the interruption is announced.
    """
    action = auto_retry_action(reason, http_status)
    if (
        action is None
        or task.state != TaskState.BLOCKED
        or task.origin == "direct_connector"
        or task.auto_retry_at is not None
    ):
        return None

    # A locked room row means a concurrent room update. Schedule on the
    # unlocked read: the due attempt checks the policy again.
    limit, _contended = await auto_retry_policy(task.room_id, db)
    if limit is None:
        return None
    if task.auto_retry_count >= limit:
        return exhausted_notice(task, reason)
    if task.stall_streak >= STALL_RESTART_THRESHOLD:
        return skipped_notice(task, reason, SKIP_NO_PROGRESS)

    attempt = task.auto_retry_count + 1
    delay = backoff_seconds(attempt, retry_after if http_status == 429 else None)
    task.auto_retry_at = datetime.now(timezone.utc) + timedelta(seconds=delay)
    task.auto_retry_reason = reason
    return AutoRetryNotice(
        TASK_AUTO_RETRY_SCHEDULED,
        f"Auto-retry {attempt}/{limit} scheduled for task \"{task.title}\".",
        {
            "attempt": attempt,
            "max_attempts": limit,
            "reason": reason,
            "action": action,
            "auto_retry_at": task.auto_retry_at.isoformat(),
        },
    )


class AutoRetryBusy(RuntimeError):
    """The room row is being updated; the attempt stays scheduled."""


async def run_claimed_auto_retry(task: Task, db: AsyncSession) -> bool:
    """Run the scheduled attempt of a task the caller has locked.

    Shared by the background cycle and the "Retry now" control, so both use
    the scheduled action (Retry or Resume) and count an automatic attempt.
    Returns True when the task was restarted. Raises :class:`AutoRetryBusy`
    after rolling back, with the schedule untouched, when the room row is
    locked by a concurrent update.
    """
    from app.core.task_restart import TaskRestartError, restart_task

    reason = task.auto_retry_reason or ""
    action = auto_retry_action(reason)
    room_id = task.room_id
    limit, contended = await auto_retry_policy(room_id, db)
    if contended:
        # Rollback expires the row; nothing below may touch its attributes.
        await db.rollback()
        raise AutoRetryBusy(str(room_id))
    task.auto_retry_at = None
    task.auto_retry_reason = None
    if action is None or task.origin == "direct_connector":
        await db.commit()
        return False
    if limit is None:
        await skipped_notice(task, reason, SKIP_ROOM_INACTIVE).emit(task, db)
        return False
    if task.auto_retry_count >= limit:
        await exhausted_notice(task, reason).emit(task, db)
        return False
    if task.stall_streak >= STALL_RESTART_THRESHOLD:
        await skipped_notice(task, reason, SKIP_NO_PROGRESS).emit(task, db)
        return False

    try:
        await restart_task(
            task, task.room_id, db, mode=action, source="auto_retry", reason=reason,
            auto_retry_limit=limit,
        )
        return True
    except TaskRestartError as exc:
        # Shared dispatch preflight checks the same hard quotas as manual
        # Retry. A refusal consumes the schedule without using an attempt.
        logger.info("auto_retry: task %s was not restarted: %s", task.id, exc)
        await skipped_notice(task, reason, SKIP_DISPATCH_REFUSED, exc.detail).emit(task, db)
        return False


async def run_due_auto_retries(db: AsyncSession, *, now: datetime | None = None) -> None:
    """Claim one due task per transaction so other workers can claim the rest."""
    for _ in range(50):
        task = (
            await db.execute(
                select(Task)
                .where(
                    Task.state == TaskState.BLOCKED,
                    Task.auto_retry_at <= (func.now() if now is None else now),
                )
                .order_by(Task.auto_retry_at, Task.id)
                .limit(1)
                .with_for_update(skip_locked=True)
                .execution_options(populate_existing=True)
            )
        ).scalar_one_or_none()
        if task is None:
            return
        try:
            await run_claimed_auto_retry(task, db)
        except AutoRetryBusy:
            # The next cycle claims it again once the room update commits.
            return
