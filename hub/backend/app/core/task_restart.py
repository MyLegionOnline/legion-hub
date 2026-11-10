"""Shared restart path for manual and automatic task recovery."""

from datetime import datetime, timezone
import logging
from typing import Literal
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auto_retry import SKIP_DISPATCH_REFUSED, RestartMode, skipped_notice
from app.core.events import emit_event
from app.core.proxy_session import (
    TaskDispatchPreflightError,
    load_paused_history,
    prepare_task_dispatch,
    request_task_dispatch,
)
from app.core.task_runtime import RUNTIME_IDLE
from app.core.watchdog import close_open_task_sessions
from app.models.message import (
    TASK_AUTO_RETRIED,
    TASK_RESUMED,
    TASK_RETRIED,
    TOOL_LIMIT_REACHED,
    Message,
)
from app.models.task import DispatchStatus, Task, TaskState

logger = logging.getLogger(__name__)


class TaskRestartError(ValueError):
    """A restart was refused before closing sessions or updating task state.

    ``detail`` is the short, user-facing cause without the endpoint wording.
    """

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.detail = detail or message


# Interruptions whose saved conversation cannot be sent again as it is: the
# prompt alone is over the cap. Resuming would fail the same way at once.
_NOT_RESUMABLE_LIMIT_REASONS = ("prompt_cap", "model_context_limit")


async def _last_limit_reason(task_id: UUID, db: AsyncSession) -> str | None:
    message = (
        await db.execute(
            select(Message.event_metadata)
            .where(Message.task_id == task_id, Message.event_type == TOOL_LIMIT_REACHED)
            .order_by(Message.timestamp.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    return (message or {}).get("reason")


async def restart_task(
    task: Task,
    room_id: UUID,
    db: AsyncSession,
    *,
    mode: RestartMode,
    source: Literal["retry", "resume", "auto_retry"],
    assigned_agent_id: UUID | None = None,
    reason: str | None = None,
    auto_retry_limit: int | None = None,
) -> RestartMode:
    """Restart an eligible task and return the mode used for dispatch.

    Callers validate eligibility and permissions before calling this function.
    """
    automatic = source == "auto_retry"
    if (
        mode == "resume"
        and not automatic
        and await _last_limit_reason(task.id, db) in _NOT_RESUMABLE_LIMIT_REASONS
    ):
        raise TaskRestartError(
            "The saved conversation is over the prompt limit and cannot be resumed — "
            "use retry instead"
        )
    if mode == "resume" and await load_paused_history(task.id, db) is None:
        if not automatic:
            raise TaskRestartError(
                "No paused limit session found for this task — use retry instead"
            )
        mode = "retry"

    original_assignee = task.assigned_agent_id
    if assigned_agent_id is not None:
        task.assigned_agent_id = assigned_agent_id

    try:
        await prepare_task_dispatch(task, db, require_ready=False)
    except TaskDispatchPreflightError as exc:
        task.assigned_agent_id = original_assignee
        action = "resumed" if mode == "resume" else "retried"
        raise TaskRestartError(
            f"Task cannot be {action}: {exc.code} ({exc})", detail=str(exc),
        ) from exc

    prev_state = task.state
    task.state = TaskState.ASSIGNED
    task.dispatch_status = DispatchStatus.READY
    if mode == "retry":
        task.pause_reason = None
        task.paused_at = None
    if automatic:
        task.auto_retry_count += 1
        task.auto_retry_reason = reason
        # Fence the interrupted run and leave the queued attempt idle until
        # the dispatcher claims it. A pause before then can settle immediately.
        task.run_id = None
        task.runtime_state = RUNTIME_IDLE
        task.stop_requested_at = None
        task.stop_reason = None
    else:
        task.auto_retry_count = 0
        task.auto_retry_reason = None
        if mode == "retry":
            task.retry_count += 1
    task.auto_retry_at = None
    started_at = datetime.now(timezone.utc).replace(tzinfo=None)
    task.updated_at = started_at

    db.add(task)
    # Closing sessions can emit an event and commit. Write the new state
    # before that first commit so the pending attempt cannot be claimed again.
    sessions_closed = await close_open_task_sessions(task.id, "retry", db)

    if automatic:
        attempt_label = str(task.auto_retry_count)
        metadata = {"attempt": task.auto_retry_count, "reason": reason, "action": mode}
        if auto_retry_limit is not None:
            attempt_label += f"/{auto_retry_limit}"
            metadata["max_attempts"] = auto_retry_limit
        await emit_event(
            room_id=room_id,
            event_type=TASK_AUTO_RETRIED,
            content=f'Auto-retry {attempt_label} for task "{task.title}" ({mode}).',
            db=db,
            task_id=task.id,
            metadata=metadata,
        )
    else:
        await db.commit()
        await db.refresh(task)
        metadata = {
            "task_id": str(task.id),
            "task_title": task.title,
            "prev_state": prev_state.value,
            "sessions_closed": sessions_closed,
        }
        if mode == "resume":
            event_type = TASK_RESUMED
            content = (
                f"Task \"{task.title}\" resumed from where it stopped "
                "(the new session's counters were reset)."
            )
        else:
            event_type = TASK_RETRIED
            content = f"Task \"{task.title}\" re-queued for retry (attempt {task.retry_count})."
            metadata["retry_count"] = task.retry_count
        await emit_event(
            room_id=room_id,
            event_type=event_type,
            content=content,
            db=db,
            task_id=task.id,
            metadata=metadata,
        )

    try:
        outcome = await request_task_dispatch(
            task, room_id, db, resume=mode == "resume", source=source,
            **({"auto_retry_started_at": started_at} if automatic else {}),
        )
    except TaskDispatchPreflightError as exc:
        if not automatic:
            raise
        # The room or quota can change after the restart was committed. Only
        # withdraw this generation; a user may already have replaced it.
        withdrawn = await db.execute(
            update(Task)
            .where(
                Task.id == task.id,
                Task.state == TaskState.ASSIGNED,
                Task.dispatch_status == DispatchStatus.READY,
                Task.updated_at == started_at,
            )
            .values(state=TaskState.BLOCKED, auto_retry_reason=None)
        )
        await db.commit()
        logger.info("auto_retry: dispatch for task %s refused: %s", task.id, exc.code)
        if withdrawn.rowcount:
            await skipped_notice(task, reason or "", SKIP_DISPATCH_REFUSED, str(exc)).emit(task, db)
        return mode
    logger.info("tasks: %s dispatch for task %s outcome=%s", mode, task.id, outcome)
    return mode
